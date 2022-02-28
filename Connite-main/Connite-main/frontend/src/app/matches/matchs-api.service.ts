import {Injectable} from '@angular/core';
import {HttpClient, HttpErrorResponse} from '@angular/common/http';
import {API_URL} from '../env';
import {Match} from './match.model';

@Injectable()
export class MatchsApiService
{
    constructor(private http: HttpClient)
    {}

    getMachtes(date:String)
    {
        return this.http
            .get<Match[]>(`${API_URL}/matchs/${date}`)
    }

}
